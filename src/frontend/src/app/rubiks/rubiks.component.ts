import { Component, OnInit } from '@angular/core';
import { CommonService } from '../common.service';

@Component({
  selector: 'app-rubiks',
  templateUrl: './rubiks.component.html',
  styles: [
    ".side { height: 100px; width: 100px; }"
  ],
})
export class RubiksComponent implements OnInit {

  constructor(public commonService: CommonService) { }

  ngOnInit(): void {
  }

}
