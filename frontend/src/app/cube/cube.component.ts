import { Component, OnInit } from '@angular/core';
import { HttpService } from '../common/http.service';
import { CommonService } from '../common/common.service';
import { CubeService } from '../common/cube.service';

@Component({
  selector: 'app-cube',
  templateUrl: './cube.component.html',
  styleUrls: ['./cube.component.css']
})
export class CubeComponent implements OnInit {

  sides = ["F", "B", "T", "D", "L", "R"];
  indices = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  colours = ["red", "orange", "white", "yellow", "green", "blue"];

  constructor(public httpService: HttpService, public commonService: CommonService, public cubeService: CubeService) { }

  ngOnInit(): void { }

  public getActionButtonClass(i: number) {
    const colour = this.colours[Math.floor(i/2)];
    return `btn btn-secondary btn-action ${colour}`;
  }

  public getStickerClass(i: number, j: number) {
    const colour = this.colours[this.cubeService.currentState69[i][j]];
    return `sticker ${colour}`;
  }
}
